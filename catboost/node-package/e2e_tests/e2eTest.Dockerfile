FROM node:20

RUN apt-get update
RUN apt-get install build-essential
RUN npm install --global verdaccio@5 npm-cli-login

ADD ./e2e_tests/node-catboost-e2e-test-package/* /var/node-catboost-e2e-test-package/
ADD ./test_data /var/node-catboost-e2e-test-package/test_data/
ADD . /var/src/

ENTRYPOINT /var/src/e2e_tests/e2e_test.sh
