export class RestClient {
  public baseUrl = "";

  public getUrl(url: string | number = "") {
    if (url) {
      return `/${this.baseUrl}/${url}`;
    }

    return `/${this.baseUrl}`;
  }
}
