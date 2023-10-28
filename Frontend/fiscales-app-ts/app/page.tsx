import Header from "./components/header";
import DNIField from "./components/forms/field";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between">
      <Header />
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        {/* Contenido de tu página */}
      </div>
      <div>
        <DNIField />
        <DNIField />
      </div>
    </main>
  );
}
