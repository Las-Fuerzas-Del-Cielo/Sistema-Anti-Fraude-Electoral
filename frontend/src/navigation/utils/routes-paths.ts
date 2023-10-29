type Routes = {
  404: string;
  root: string;
  AUTH: {
    login: string;
  };
  USER: {
    profile: string;
    send_success: string;
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
    send_success: "/user/send-success",
  },
};
