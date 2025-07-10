from torch.utils.data import DataLoader
from imports.mdm.data_loaders.tensors import collate as all_collate
from imports.mdm.data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from imports.mdm.data_loaders.a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from imports.mdm.data_loaders.a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from imports.mdm.data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from imports.mdm.data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from imports.mdm.data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', david_dataset="ComAsset", david_category="frypan"):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, david_dataset=david_dataset, david_category=david_category)
    else:
        dataset = DATA(split=split, num_frames=num_frames, david_dataset=david_dataset, david_category=david_category)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', david_dataset="ComAsset", david_category="frypan"):
    dataset = get_dataset(name, num_frames, split, hml_mode, david_dataset, david_category)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader