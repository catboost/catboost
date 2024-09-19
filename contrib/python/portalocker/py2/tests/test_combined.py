import sys


def test_combined(tmpdir):
    from distutils import dist
    import setup

    output_file = tmpdir.join('combined.py')
    combine = setup.Combine(dist.Distribution())
    combine.output_file = str(output_file)
    combine.run()
    sys.path.append(output_file.dirname)
    import combined
    assert combined

