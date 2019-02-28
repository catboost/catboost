PY3TEST()



ENV(LC_ALL=ru_RU.UTF-8)
ENV(LANG=ru_RU.UTF-8)
# because we cannot change TZ in arcadia CI
ENV(DATEUTIL_MAY_NOT_CHANGE_TZ_VAR=1)

PEERDIR(
    contrib/python/dateutil/tests
)

NO_LINT()

END()
