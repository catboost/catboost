#/bin/sh
echo -n '# ' > revisions.txt
date >> revisions.txt
echo -n '# Trunk revision: ' >> revisions.txt
svn info --show-item revision svn+ssh://arcadia.yandex.ru/arc/trunk >> revisions.txt
svn ls -R -v svn+ssh://arcadia.yandex.ru/arc/trunk/arcadia_tests_data | awk -F ' ' '{print $1, $NF}'  >> revisions.txt
