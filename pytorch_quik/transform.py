import torch
import pytorch_quik.utils as pu


def tensor_dataset(tens, dt, args, ttype='train', write=True):
    for i, col in enumerate(tens):
        tens[col] = tens[col].astype(dt[i])
    ttar = tens.pop('rating')
    tens = torch.tensor(tens.values)
    tens = tens.transpose(0, 1)
    ttar = torch.tensor(ttar.values)
    tds = torch.utils.data.TensorDataset(*tens, ttar)
    if write:
        tds_id = pu.id_str(ttype, args)
        torch.save(tds, tds_id)
    return tds
