import torch
from tqdm import tqdm
from functions.svd_operators import Inpainting


class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def get_l2_similarity_matrix(x, sqrt=False):
    temp = torch.mm(x.view(x.shape[0], -1), x.view(x.shape[0], -1).t())
    diag = temp.diag().unsqueeze(0)
    diag = diag.expand_as(temp)
    D = diag + diag.t() - 2*temp
    if sqrt:
        D = D.sqrt()
    return D


def get_nn(x, missing_pixs):
    masked_x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1).reshape(x.shape[0], -1)[:, missing_pixs]
    similarity_mat = get_l2_similarity_matrix(masked_x)
    _, nn_i = torch.topk(similarity_mat, 2, largest=False)
    # print(f"Debug: {nn_i[:,1]}")
    nn_x = x[nn_i[:, 1], :, :, :]
    return nn_x, nn_i[:, 1]


def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None,
                   guidance_eta=0, g_dist=None):
    with torch.no_grad():

        inpainting = type(A_funcs) == Inpainting
        mask_size = len(A_funcs.missing_indices) if inpainting else x[0].numel()

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        x0_orig_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling,
                                  config.time_travel.travel_length,
                                  config.time_travel.travel_repeat,
                                  )
        time_pairs = list(zip(times[:-1], times[1:]))

        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j < 0:
                j = -1

            if j < i:  # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * classes.cuda()
                    # classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # # ---- 4guidance
                if guidance_eta != 0 and j != -1:
                    guidance_factor = j / config.diffusion.num_diffusion_timesteps

                    if inpainting:
                        x0_4direction, nn_idx = get_nn(x0_t, A_funcs.missing_indices)
                    else:
                        x0_4direction, nn_idx = get_nn(x0_t, None)

                    dist_f = torch.ones(x0_t.shape[0], 1, 1, 1).to(x0_t.device)
                    if g_dist is not None:
                        distance = torch.linalg.norm((x0_t - x0_4direction).reshape(x0_t.shape[0], -1), dim=1
                                                     ) / mask_size
                        # Hard thresholds
                        # don't change images where normalized distance is larger than X
                        dist_f[distance > g_dist] = 0
                    x0_t_g = x0_t + (x0_t - x0_4direction) * guidance_eta * guidance_factor * dist_f
                    x0_orig_preds.append(x0_t.to('cpu'))
                    x0_t = x0_t_g
                # # --------------

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else:         # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')

                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]], x0_preds, x0_orig_preds, xs


def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None,
                        guidance_eta=0, g_dist=None):
    with torch.no_grad():

        inpainting = type(A_funcs) == Inpainting
        mask_size = len(A_funcs.missing_indices) if inpainting else x[0].numel()

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling,
                                  config.time_travel.travel_length,
                                  config.time_travel.travel_repeat,
                                  )
        time_pairs = list(zip(times[:-1], times[1:]))

        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j < 0:
                j = -1

            if j < i:  # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * classes.cuda()
                    # classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # # ---- 4guidance
                if guidance_eta != 0 and j != -1:
                    guidance_factor = j / config.diffusion.num_diffusion_timesteps

                    if inpainting:
                        x0_4direction, nn_idx = get_nn(x0_t, A_funcs.missing_indices)
                    else:
                        x0_4direction, nn_idx = get_nn(x0_t, None)

                    dist_f = torch.ones(x0_t.shape[0], 1, 1, 1).to(x0_t.device)
                    if g_dist is not None:
                        distance = torch.linalg.norm((x0_t - x0_4direction).reshape(x0_t.shape[0], -1), dim=1
                                                     ) / mask_size
                        # Hard thresholds
                        # don't change images where normalized distance is larger than X
                        dist_f[distance > g_dist] = 0
                    x0_t_g = x0_t + (x0_t - x0_4direction) * guidance_eta * guidance_factor * dist_f
                    x0_t = x0_t_g
                # # --------------

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else:  # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')

                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
