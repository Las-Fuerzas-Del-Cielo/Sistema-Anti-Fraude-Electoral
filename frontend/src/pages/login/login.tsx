import Button from "#/components/button";
import Input from "#/components/input";
import Footer from "#/components/footer";
import { useFormik } from "formik";
import { observer } from "mobx-react-lite";

interface ILoginProps {
  dni: string;
  password: string;
}

const LoginPage = () => {
  const onSubmit = (values: ILoginProps) => {
    console.log(values);
  };

  const { handleSubmit, handleChange } = useFormik({
    initialValues: {
      dni: "",
      password: "",
    },
    onSubmit,
  });

  return (
    <section className="bg-gray-100 h-screen overflow-hidden items-center flex flex-col relative">
    <div className="md:w-1/2 w-5/6 shadow-3xl rounded-xl p-4 z-10">
        <div className="container mx-auto">
          <div className="flex items-center justify-center my-20">
            <img
              src="src/assets/logos/fenix.png"
              alt="fenix"
              className="object-cover rounded w-28 h-auto mr-4"
            />
            <img
              src="src/assets/logos/lla.svg"
              alt="lla"
              className="object-cover rounded w-50 h-auto"
            />
          </div>
        </div>
        <form className="w-full" onSubmit={handleSubmit}>
          <div className="flex items-center text-lg mb-6 md:mb-8 shadow-3xl">
            <Input
              label="DNI"
              type="text"
              id="dni"
              className="bg-gray-200 rounded-xl pl-4 py-4 focus:outline-none w-full border-2 border-gray-300"
              placeholder="Ingresa tu DNI"
              onChange={handleChange}
            />
          </div>
          <div className="flex items-center text-lg mb-6 md:mb-8 shadow-3xl">
            <Input
              label="Contraseña"
              type="password"
              id="password"
              className="bg-gray-200 rounded-xl pl-4 py-4 focus:outline-none w-full border-2 border-gray-300"
              placeholder="Ingresa tu Contraseña"
              onChange={handleChange}
            />
          </div>
          <div className="flex flex-col items-center text-lg">
            <Button
              className="bg-violet-brand p-4 text-white w-full rounded-xl font-semibold text-xl tracking-wider"
              type="submit"
              label="Ingresar"
            />
            <a
              href="#"
              className="text-center text-gray-600 text-lg underline mt-8"
            >
              ¿Necesitas ayuda?
            </a>
          </div>
        </form>
        
      </div>
      <Footer />
    </section>
  );
};

export const Login = observer(LoginPage);

export default Login;
