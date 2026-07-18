import unittest
from unittest import mock

from build import build_native


class TestSanitizerCmakeArgs(unittest.TestCase):
    def make_opts(self, **kwargs):
        return build_native.Opts(build_root_dir='build', targets=['catboost'], **kwargs)

    def test_asan_and_ubsan_flags(self):
        cmake_args = build_native.get_sanitizer_cmake_args(
            self.make_opts(with_asan=True, with_ubsan=True)
        )

        self.assertEqual(
            cmake_args,
            [
                '-DCUSTOM_ALLOCATORS=Off',
                '-DCMAKE_C_FLAGS_INIT=-fsanitize=address,undefined -fno-omit-frame-pointer',
                '-DCMAKE_CXX_FLAGS_INIT=-fsanitize=address,undefined -fno-omit-frame-pointer'
            ]
        )

    def test_tsan_cannot_be_combined_with_asan(self):
        with self.assertRaisesRegex(RuntimeError, 'ThreadSanitizer cannot be combined'):
            build_native.get_sanitizer_cmake_args(self.make_opts(with_asan=True, with_tsan=True))

    def test_sanitizers_cannot_be_combined_with_cuda(self):
        with self.assertRaisesRegex(RuntimeError, 'Sanitizers are not supported together with CUDA builds'):
            build_native.get_sanitizer_cmake_args(self.make_opts(with_asan=True, have_cuda=True))

    @mock.patch.object(build_native.platform, 'system', return_value='Windows')
    def test_windows_asan_does_not_use_unix_frame_pointer_flag(self, _platform_system):
        cmake_args = build_native.get_sanitizer_cmake_args(self.make_opts(with_asan=True))

        self.assertEqual(
            cmake_args,
            [
                '-DCUSTOM_ALLOCATORS=Off',
                '-DCMAKE_C_FLAGS_INIT=-fsanitize=address',
                '-DCMAKE_CXX_FLAGS_INIT=-fsanitize=address'
            ]
        )

    @mock.patch.object(build_native.platform, 'system', return_value='Windows')
    def test_windows_tsan_is_not_supported(self, _platform_system):
        with self.assertRaisesRegex(RuntimeError, 'ThreadSanitizer is not supported on Windows'):
            build_native.get_sanitizer_cmake_args(self.make_opts(with_tsan=True))


if __name__ == '__main__':
    unittest.main()
