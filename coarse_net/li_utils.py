import torch
def skew_sym_mat(x):# 生成3维向量对应的反对称矩阵
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):# 指数映射函数将李代数中的向量（这里是theta）映射到李群（SO(3)）中的一个旋转矩阵
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta) # 其中theta为旋转向量，W为旋转向量的斜对称矩阵
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    #  当theta的范数（即角度）小于1e-5时，使用近似展开式：
    #   exp(theta) ≈ I + W + 0.5 * W^2
    else:
    # 当theta的范数大于等于1e-5时，使用精确的公式：其中angle为theta的范数
    #   exp(theta) = I + (sin(angle) / angle) * W + ((1 - cos(angle)) / angle^2) * W^2
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):#计算SE3李代数指数映射的雅可比矩阵部分
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):# tau为SE3的六维向量 tau = [rho theta]
    # 指数映射函数将李代数中的向量（tau）映射到李群（SE(3)）中的一个变换矩阵
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta) #计算SO3部分的旋转矩阵
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T