#!/bin/sh
verdaccio &
sleep 2
npm-cli-login -u test_user -p test_password -e test@example.com@example.com -r http://localhost:4873 && \
    cd /var/src && \
    npm publish --registry http://localhost:4873 && \
    cd /var/node-catboost-e2e-test-package && \
    npm install catboost --registry http://localhost:4873 && \
    node e2e_predict.js > /tmp/out.log && \
    diff /tmp/out.log ./expected.log
