/* eslint-disable react-refresh/only-export-components */
import { RootScreen } from "#/pages";
import {
  Outlet,
  RouteObject,
  RouterProvider,
  createBrowserRouter,
} from "react-router-dom";
import { ROUTES_PATHS, authRoutes, userRoutes } from "./utils";

export const appRoutes: RouteObject[] = [
  {
    path: ROUTES_PATHS[404],
    element: <p>Not found page</p>,
    errorElement: <p>Error page</p>,
  },
  {
    path: ROUTES_PATHS.root,
    element: <RootScreen />,
    errorElement: <p>Error page</p>,
    children: [
      {
        path: ROUTES_PATHS.AUTH.login,
        element: <Outlet />,
        children: authRoutes,
      },

      {
        path: ROUTES_PATHS.USER.profile,
        element: <Outlet />,
        children: userRoutes,
      },
    ],
  },
];

const router = createBrowserRouter(appRoutes);

const Routes = () => (
  <RouterProvider router={router} fallbackElement={<p>Loading...</p>} />
);

export default Routes;
