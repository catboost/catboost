import sys

from jinja2 import Environment, PackageLoader


env = Environment(
    loader=PackageLoader("hello", "templates"),
)

username = sys.argv[1] if len(sys.argv) > 1 else "Anonymous"

print(env.get_template("hello").render(username=username))
