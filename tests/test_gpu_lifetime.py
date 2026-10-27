import numpy as np
import pytest
from stereoforge.gpu import VulkanStereo

def test_pending_submission_cannot_be_overwritten_and_foreign_token_rejected():
    gpu=VulkanStereo(8)
    image=np.arange(8*12,dtype=np.uint32).reshape(8,12)
    token=gpu.enqueue(image,image)
    with pytest.raises(RuntimeError):gpu.enqueue(image,image)
    with pytest.raises(ValueError):gpu.result(token+1)
    gpu.result(token)
    with pytest.raises(ValueError):gpu.result(token)
    gpu.result(gpu.enqueue(image,image))

pytestmark = pytest.mark.gpu
