import { useAuth } from "#/hooks";
import { observer } from "mobx-react-lite";
import { FC } from "react";
import { Outlet } from "react-router-dom";

const RootScreenContent: FC = () => {
  useAuth();
  return <Outlet />;
};

export const RootScreen = observer(RootScreenContent);
