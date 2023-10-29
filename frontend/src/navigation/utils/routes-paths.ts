type Routes = {
  404: string;
  root: string;
  AUTH: {
    login: string;
  };
  USER: {
    profile: string;
  };
};

export const ROUTES_PATHS: Routes = {
  404: "*",
  root: "/",
  AUTH: {
    login: "/auth/login",
  },
  USER: {
    profile: "/user/profile",
  },
};
