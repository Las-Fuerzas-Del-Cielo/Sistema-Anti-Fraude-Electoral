import { makeAutoObservable, runInAction } from "mobx";

import { RootStore } from "./root";

export class AuthStore {
  public readonly rootStore: RootStore;

  public isLoggedIn: boolean = false;

  constructor(rootStore: RootStore) {
    makeAutoObservable(this);

    this.rootStore = rootStore;
  }

  public startLogin = () => {
    // Cambiar este boolean para manejar el estado de login/logout hasta tener la API.
    const loginApiResponse = false;

    runInAction(() => {
      this.isLoggedIn = loginApiResponse;
    });
  };
}
