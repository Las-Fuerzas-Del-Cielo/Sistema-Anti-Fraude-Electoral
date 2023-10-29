import Button from "#/components/button";
import Input from "#/components/input";
import { useFormik } from "formik";
import { observer } from "mobx-react-lite";

interface ILoginProps {
  dni: string;
  password: string;
}

const LoginPage = () => {
  return (
    <section className="relative flex flex-col items-center h-screen gap-4 p-6 overflow-hidden bg-gray-100">
      <Input placeholder="Input de texto" id="text" type="text" label="Input de texto" onChange={() => {}} />
      <Input placeholder="Input de password" id="password" type="password" label="Input de password" onChange={() => {}} />
      <Input placeholder="Input underline" id="underline" type="text" label="Input underline" appearance="underline" onChange={() => {}} />
      <Input placeholder="Input error" id="error" type="text" label="Input error" error onChange={() => {}} />
    </section>
  );
};

export const Login = observer(LoginPage);

export default Login;
