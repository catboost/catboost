import optparse


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest='file_path')
    parser.add_option('-a', '--append', action='store_true', default=False)
    parser.add_option('-Q', '--quote', action='store_true', default=False)
    parser.add_option('-s', '--addspace', action='store_true', default=False)
    parser.add_option('-c', '--content', dest='content')
    return parser.parse_args()


def smart_shell_quote(v):
    if v is None:
        return None
    if ' ' in v or '"' in v or "'" in v:
        return "\"{0}\"".format(v.replace('"', '\\"'))
    return v

if __name__ == '__main__':
    opts, _ = parse_args()
    open_type = 'a' if opts.append else 'w'

    content = opts.content
    if opts.quote:
        content = smart_shell_quote(content)

    with open(opts.file_path, open_type) as f:
        if opts.addspace:
            f.write(' ')
        if content is not None:
            f.write(content)
