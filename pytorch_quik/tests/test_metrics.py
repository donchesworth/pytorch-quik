import pytorch_quik as pq
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd
import pytest


def test_dir_classes(senti_classes):
    raw_classes = list(senti_classes.values())
    dir_classes = pq.metrics.direct_dict(raw_classes)
    assert isinstance(dir_classes, OrderedDict)
    assert dir_classes == senti_classes


def test_inv_classes(senti_classes, inv_senti_classes):
    inv_classes = pq.metrics.inverse_dict(senti_classes)
    assert isinstance(inv_classes, OrderedDict)
    assert inv_classes == inv_senti_classes


def test_choose_a_class():
    cc = torch.rand(100, 100)
    cc = pq.metrics.choose_a_class(cc)
    assert cc.shape == torch.Size([100])


def test_numpize_array(args):
    ln = 100
    wd = 3
    t = torch.rand(ln, wd)
    p = pd.DataFrame(np.random.randint(0, ln, size=(ln, wd)))
    s = pd.Series(np.random.randint(0, ln, size=ln))
    objs = [t, p, s]
    if bool(args.gpus) and not args.has_gpu:
        with pytest.raises(ImportError):
            import cudf
    elif args.gpus == 1:
        import cudf
        c = cudf.DataFrame.from_pandas(p)
        objs.append(c)
    assert all(
        [isinstance(pq.metrics.numpize_array(x), np.ndarray) for x in objs]
    )


def test_show_confusion_matrix(
    final_cmdf, sample_preds, sample_labels, senti_classes
):
    cm = pq.metrics.show_confusion_matrix(
        sample_preds, sample_labels, senti_classes)
    assert cm.equals(final_cmdf)


@pytest.mark.mpl_image_compare
def test_matplotlib_confusion_matrix(final_cmdf, senti_classes):
    npdf = final_cmdf.to_numpy()
    cm = pq.metrics.matplotlib_confusion_matrix(npdf, senti_classes)
    return cm
