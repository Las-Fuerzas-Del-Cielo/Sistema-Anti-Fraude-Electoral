FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY package*.json ./

RUN yarn install --frozen-lockfile

COPY . /app

EXPOSE 9000
ENV HOST=0.0.0.0
ENV PORT=9000

RUN if [ "$VERSION" ];then npm i -g serve && npm run build; fi
