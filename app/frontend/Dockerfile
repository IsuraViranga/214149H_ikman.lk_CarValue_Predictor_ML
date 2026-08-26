# ---- build the React bundle ----
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# CRA inlines env vars at build time, so the API URL is a build arg
ARG REACT_APP_API_URL=http://localhost:5000/api
ENV REACT_APP_API_URL=$REACT_APP_API_URL
RUN npm run build

# ---- serve it ----
FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
