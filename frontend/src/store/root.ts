import axios from "axios";
import { UiStore } from "./ui";

//aca configuras la url de la api
const API_BASE_URL = "www.asda.com";

export class RootStore {
  public readonly uiStore;

  constructor() {
    axios.defaults.baseURL = API_BASE_URL;

    this.uiStore = new UiStore(this);
  }
}

export const store = new RootStore();
