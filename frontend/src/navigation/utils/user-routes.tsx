import { Profile } from "#/pages";
import { RouteObject } from "react-router-dom";
import { ROUTES_PATHS } from "./routes-paths";

export const userRoutes: RouteObject[] = [
  {
    path: ROUTES_PATHS.USER.profile,
    element: <Profile />,
  },
];
