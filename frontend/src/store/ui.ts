import { makeAutoObservable, runInAction } from "mobx";

import { RootStore } from "./root";
import { testClient } from "#/service";

export class UiStore {
  public readonly rootStore: RootStore;

  public globalTestid: string = "algo";

  constructor(rootStore: RootStore) {
    makeAutoObservable(this);

    this.rootStore = rootStore;
  }

  public globalTestIdAction = (nuevoTexto: string) => {
    runInAction(() => {
      this.globalTestid = nuevoTexto;
    });
  };

  public nuevaFuncion = async () => {
    const response = await testClient.apiCallTest();

    runInAction(() => {
      this.globalTestid = response;
    });
  };
}
