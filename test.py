import torch.utils.cpp_extension


if __name__ == '__main__':
    # Load file
    # TODO: change directory into an arg or tmp directory
    with open('cmake-build-debug/AddOp.pytorch.h') as file:
        src = file.read()

    # Compile
    module = torch.utils.cpp_extension.load_inline(
        name='test_inline_ext',
        cpp_sources=[src],
        functions=['AddOp_th_'],
        extra_cflags=['-Wno-c++17-extensions', '-std=c++14', '-g'],
        # TODO: change directory into an arg or tmp directory
        extra_ldflags=['-L/Users/lyricz/Desktop/Projects/HyTorch/cmake-build-debug',
                       '-lAddOp', '-lAddOp.runtime'],
        verbose=True
    )

    # Test
    # TODO: may warp below functions into a PyTorch module
    shape = [32, 32, 32, 32]
    x, y, z = torch.randn(shape), torch.randn(shape), torch.zeros(shape)
    module.AddOp_th_(x, y, z)
    assert (x + y - z).sum().item() < 1e-6
