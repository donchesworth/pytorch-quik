from pytorch_quik import ddp


def test_tq_bar():
    size = 5
    pbar = ddp.tq_bar(size)
    for _ in range(size):
        pbar.update()
    pbar.close()
    del pbar


def test_find_free_port():
    port = ddp.find_free_port()
    assert isinstance(port, int)
