import pytorch_quik as pq
import torch
import numpy as np
import pandas as pd
import pytest


def test_choose_a_class():
    cc = torch.rand(100, 100)
    cc = pq.metrics.choose_a_class(cc)
    assert cc.shape == torch.Size([100])


@pytest.mark.skip_gpus
def test_numpize_array(args):
    ln = 100
    wd = 3
    t = torch.rand(ln, wd)
    p = pd.DataFrame(np.random.randint(0, ln, size=(ln, wd)))
    s = pd.Series(np.random.randint(0, ln, size=ln))
    objs = [t, p, s]
    if args.has_gpu:
        import cudf
        c = cudf.DataFrame.from_pandas(p)
        objs.append(c)
    assert all(
        [isinstance(pq.metrics.numpize_array(x), np.ndarray) for x in objs]
    )


def test_build_confusion_matrix(
    final_cmdf, sample_preds, sample_labels, senti_classes
):
    cm = pq.metrics.build_confusion_matrix(
        sample_preds, sample_labels, senti_classes)
    assert np.array_equal(cm, final_cmdf)


# @pytest.mark.mpl_image_compare
# def test_matplotlib_confusion_matrix(final_cmdf, senti_classes):
#     npdf = final_cmdf.to_numpy()
#     cm = pq.metrics.matplotlib_confusion_matrix(npdf, senti_classes)
#     return cm
