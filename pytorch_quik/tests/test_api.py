import pytorch_quik as pq
import numpy as np
import dask_quik as dq
import pytest

BATCH_SIZE = 3
output = '{"instances":[{"data": "this is absolutely terrible"}, {"data": '


def test_split_and_format(sample_data):
    txt_list = pq.api.split_and_format(
        sample_data.to_numpy()[:, 0],
        BATCH_SIZE
        )
    output = '{"instances":[{"data": "this is absolutely terrible"}, {"data":'
    assert (txt_list[0][:63] == output)


def test_batch_inference(sample_data, senti_classes):
    if dq.utils.gpus() == 0:
        print("unable to test batch_inference on internal API")
        pytest.skip()
    output = np.array([
        'Positive', 'Positive', 'Positive', 'Positive', 'Neutral',
        'Negative', 'Positive', 'Positive'], dtype=object)
    rdf = pq.api.batch_inference(
        sample_data.to_numpy()[:, 0], senti_classes, BATCH_SIZE
        )
    assert np.array_equal(output, rdf['label'].values)
