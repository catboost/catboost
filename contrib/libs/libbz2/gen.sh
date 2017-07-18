#1 /bin/sh
cat *.c *.h | grep BZ_API | sed -e 's/.*BZ_API//' | grep BZ | sed -e 's/ .*//'  |sed -e 's/(//' | sed -e 's/)//' | sort |uniq | grep -v '(' | while read l; do echo "#define $l Arc$l"; done
