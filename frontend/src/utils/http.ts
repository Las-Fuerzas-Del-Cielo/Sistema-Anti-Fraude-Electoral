import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';

import { eventBus } from './event-bus';
import { sleep } from './utils';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Data = Record<string, any>;

export class Http {
  public isInternetReachable = true;
  private waitForConnectionPromise: Promise<void> | undefined;

  public constructor() {
    axios.defaults.baseURL = '/api';
  }

  public setConnection(isInternetReachable: boolean) {
    this.isInternetReachable = isInternetReachable;
  }

  public async waitForConnection(maxRetries = 10) {
    if (this.isInternetReachable) {
      return;
    }

    if (this.waitForConnectionPromise) {
      return this.waitForConnectionPromise;
    }

    for (let index = 0; index <= maxRetries; index++) {
      this.waitForConnectionPromise = sleep(1000);

      await this.waitForConnectionPromise;

      this.waitForConnectionPromise = undefined;

      if (this.isInternetReachable) {
        return;
      }
    }
  }

  public async get(path: string, options: AxiosRequestConfig = {}) {
    await this.waitForConnection().catch();

    return axios.get(path, options).then(this.mapData).catch(this.handleError);
  }

  public async post(path: string, data?: Data, options: AxiosRequestConfig = {}) {
    await this.waitForConnection().catch();

    return axios.post(path, data, options).then(this.mapData).catch(this.handleError);
  }

  public async put(path: string, data?: Data, options: AxiosRequestConfig = {}) {
    await this.waitForConnection().catch();

    return axios.put(path, data, options).then(this.mapData).catch(this.handleError);
  }

  public async patch(path: string, data?: Data, options: AxiosRequestConfig = {}) {
    await this.waitForConnection().catch();

    return axios.patch(path, data, options).then(this.mapData).catch(this.handleError);
  }

  public async delete(path: string, options: AxiosRequestConfig = {}) {
    await this.waitForConnection().catch();

    return axios.delete(path, options).then(this.mapData).catch(this.handleError);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private mapData = (result: AxiosResponse<any>) => {
    const error = result?.data?.errors?.[0]?.extensions?.code;

    const hasErrors = result?.data?.errors?.length > 0;

    if (hasErrors) {
      if (error === 'UNAUTHENTICATED') {
        eventBus.emit('UNAUTHORIZED');
      } else {
        this.handleError(result?.data?.errors[0]);
      }
    }
    return result.data;
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private handleError = (error: AxiosError<any>) => {
    if (error.response?.status === 401) {
      eventBus.emit('UNAUTHORIZED');
    }

    throw error;
  };
}

export const http = new Http();
