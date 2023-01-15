FROM node:15.14.0-alpine3.10

RUN apk add curl g++ libc6-compat make python3
RUN ln -s /lib/libc.musl-x86_64.so.1 /lib/ld-linux-x86-64.so.2
RUN npm install --global verdaccio npm-cli-adduser

