import string
import random


def uniq_string_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return lambda: ''.join(random.choice(chars) for _ in range(size))


gen4 = uniq_string_generator(4)
gen8 = uniq_string_generator(8)
gen16 = uniq_string_generator(16)
gen32 = uniq_string_generator(32)
gen64 = uniq_string_generator(64)
