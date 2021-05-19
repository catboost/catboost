FROM ikholopov/catboost-node-package-e2e-base:latest

ADD ./e2e_tests/node-catboost-e2e-test-package/* /var/node-catboost-e2e-test-package/
ADD ./test_data /var/node-catboost-e2e-test-package/test_data/
ADD . /var/src/

RUN chmod +x /var/src/e2e_tests/e2e_test.sh

ENTRYPOINT /var/src/e2e_tests/e2e_test.sh
