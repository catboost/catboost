#!/bin/sh
verdaccio &
sleep 2
npm-cli-adduser -u test_user -p test_password -e test@example.com@example.com -r http://0.0.0.0:4873 && \
    cd /var/src && \
    npm publish --registry http://0.0.0.0:4873 && \
    cd /var/node-catboost-e2e-test-package && \
    npm install catboost --registry http://0.0.0.0:4873 && \
    node e2e_predict.js > /tmp/out.log && \
    diff /tmp/out.log ./expected.log