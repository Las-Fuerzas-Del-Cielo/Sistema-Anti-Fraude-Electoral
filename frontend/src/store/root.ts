import axios from "axios";
import { UiStore } from "./ui";
import { AuthStore } from "./auth";

//aca configuras la url de la api
const API_BASE_URL = "www.asda.com";

export class RootStore {
  public readonly uiStore;
  public readonly authStore;

  constructor() {
    axios.defaults.baseURL = API_BASE_URL;

    this.uiStore = new UiStore(this);
    this.authStore = new AuthStore(this);
    this.login();
  }

  private login = () => {
    this.authStore.startLogin();
  };
}

export const store = new RootStore();
