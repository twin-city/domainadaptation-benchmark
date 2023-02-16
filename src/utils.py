# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import os.path as osp
import requests
import warnings


def download_checkpoint(checkpoint_name, model_name, config_name, collect_dir):
    """s
    Copied from mmsegmentation .dev
    Download checkpoint and check if hash code is true."""
    url = f'https://download.openmmlab.com/mmsegmentation/v0.5/{model_name}/{config_name}/{checkpoint_name}'  # noqa

    r = requests.get(url)
    assert r.status_code != 403, f'{url} Access denied.'

    with open(osp.join(collect_dir, checkpoint_name), 'wb') as code:
        code.write(r.content)

    try:
        true_hash_code = osp.splitext(checkpoint_name)[0].split('-')[1]

        # check hash code
        with open(osp.join(collect_dir, checkpoint_name), 'rb') as fp:
            sha256_cal = hashlib.sha256()
            sha256_cal.update(fp.read())
            cur_hash_code = sha256_cal.hexdigest()[:8]

        assert true_hash_code == cur_hash_code, f'{url} download failed, '
        'incomplete downloaded file or url invalid.'

        if cur_hash_code != true_hash_code:
            os.remove(osp.join(collect_dir, checkpoint_name))
    except AssertionError:
        warnings.warn("Hash code checking failed")