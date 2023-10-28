FROM node:18-alpine as dev

# Set NODE_ENV in development mode
ENV NODE_ENV development

# Install missing packages
RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package*.json ./

# Install dependencies
RUN yarn install --frozen-lockfile

# Create non-root user for Docker
RUN addgroup --system --gid 1001 node
RUN adduser --system --uid 1001 node

# Copy source code into app folder
COPY --chown=node:node . .

# Set Docker as a non-root user
USER node

EXPOSE 8080
