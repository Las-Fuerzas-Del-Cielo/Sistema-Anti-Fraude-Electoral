import axios, { AxiosResponse, AxiosError, Method } from 'axios';

const API_URL = 'http://localhost:5000';

type T = any;

interface FetchResponse<T> {
  data?: T;
  error?: AxiosError;
  loading: boolean;
}

const useAxios = () => {
  const fetchData = async (
    method: Method,
    url: string,
    body?: any,
  ): Promise<FetchResponse<T>> => {
    let responseState: FetchResponse<T> = {
      loading: true,
      data: undefined,
      error: undefined,
    };

    try {
      const response: AxiosResponse<T> = await axios({
        method,
        url: `${API_URL}${url}`,
        data: body,
      });

      if (response.status >= 400) throw new Error(response.statusText);

      responseState = {
        ...responseState,
        data: response.data,
        loading: false,
      };
    } catch (error) {
      responseState = {
        ...responseState,
        loading: false,
        error: error as AxiosError,
      };
    }

    return responseState;
  };

  const get = (url: string) => fetchData('GET', url);
  const post = (url: string, body: any) => fetchData('POST', url, body);
  const put = (url: string, body: any) => fetchData('PUT', url, body);
  const del = (url: string) => fetchData('DELETE', url);

  return {
    get,
    post,
    put,
    delete: del,
  };
};

export default useAxios;
