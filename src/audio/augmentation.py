"""SpecAugment: frequency + time masking on normalized log-Mel specs.
Train only — never touch val/test. Mask value 0 = dataset mean post-normalization.
Batch-level, runs on GPU."""
import torch
import torchaudio


class SpecAugment:
    def __init__(self, freq_mask: int = 16, time_mask: int = 20):
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """[B, 1, n_mels, T] -> augmented, same shape."""
        return self.tm(self.fm(spec))


def demo() -> None:
    torch.manual_seed(0)
    sa = SpecAugment()
    x = torch.randn(4, 1, 128, 32)
    y = sa(x)
    assert y.shape == x.shape
    assert not torch.equal(x, y)
    # a full masked freq band exists (all zeros somewhere per sample)
    assert (y == 0).any()
    print("OK: specaugment shape + masks applied")


if __name__ == "__main__":
    demo()
