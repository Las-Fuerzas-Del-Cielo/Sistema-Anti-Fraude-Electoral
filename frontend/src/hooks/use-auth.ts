import { ROUTES_PATHS } from "#/navigation";
import { useStore } from "#/store";
import { useCallback, useEffect, useMemo } from "react";
import { useNavigation } from "./utils";
import { useLocation } from "react-router-dom";

export const useAuth = () => {
  const { authStore } = useStore();
  const navigation = useNavigation();
  const location = useLocation();

  const authPath = useMemo(
    () => location.pathname.split("/").slice(0, 2).join("/"),
    [location.pathname]
  );
  const handleAuth = useCallback(() => {
    if (authStore.isLoggedIn) {
      if (authPath !== ROUTES_PATHS.USER.profile) {
        return navigation.replace(ROUTES_PATHS.USER.profile);
      }
      return;
    }
    if (authPath !== ROUTES_PATHS.AUTH.login) {
      return navigation.replace(ROUTES_PATHS.AUTH.login);
    }
  }, [authPath, authStore.isLoggedIn, navigation]);

  useEffect(() => {
    handleAuth();
  }, [navigation, handleAuth]);
};
