import { RouteObject } from "react-router-dom";
import { ROUTES_PATHS } from "./routes-paths";
import { Login } from "#/pages";

export const authRoutes: RouteObject[] = [
  {
    path: ROUTES_PATHS.AUTH.login,
    element: <Login />,
  },
];
