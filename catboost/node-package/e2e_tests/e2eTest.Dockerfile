FROM node:15.14.0-alpine3.10

RUN apk add curl g++ libc6-compat make python3
RUN ln -s /lib/libc.musl-x86_64.so.1 /lib/ld-linux-x86-64.so.2
RUN npm install --global verdaccio npm-cli-adduser

ADD ./e2e_tests/node-catboost-model-e2e-test-package/* /var/node-catboost-model-e2e-test-package/
ADD ./test_data /var/node-catboost-model-e2e-test-package/test_data/
ADD . /var/src/

RUN chmod +x /var/src/e2e_tests/e2e_test.sh

ENTRYPOINT /var/src/e2e_tests/e2e_test.sh
