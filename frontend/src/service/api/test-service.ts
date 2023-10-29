import { http } from '#/utils';
import { RestClient } from '../rest';

class TestClient extends RestClient {
  public baseUrl = '/api';

  public async apiCallTest() {
    const response = await http.get(this.getUrl(`cualquiercosa`));

    return response.data;
  }
}

export const testClient = new TestClient();
