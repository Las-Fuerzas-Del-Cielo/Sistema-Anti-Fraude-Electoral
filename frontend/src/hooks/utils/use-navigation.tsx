import { useCallback } from "react";
import { useNavigate } from "react-router-dom";

export const useNavigation = () => {
  const navigate = useNavigate();

  const push = useCallback(
    (route: string) => {
      navigate(route);
    },
    [navigate]
  );

  const replace = useCallback(
    (route: string) => {
      navigate(route, {
        replace: true,
      });
    },
    [navigate]
  );

  const pop = useCallback(
    (pagesAmount?: number) => {
      const pages = pagesAmount ? -pagesAmount : -1;
      navigate(pages);
    },
    [navigate]
  );

  const popToTop = useCallback(() => {
    navigate("/", {
      replace: true,
    });
  }, [navigate]);

  const refresh = useCallback(() => {
    navigate(0);
  }, [navigate]);

  return { push, pop, replace, popToTop, refresh };
};
