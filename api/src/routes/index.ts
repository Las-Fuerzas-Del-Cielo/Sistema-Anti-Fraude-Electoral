import denunciaPath from "./paths/denuncia.path";
import fiscalizarPath from "./paths/fiscalizar.path";
import mesaPath from "./paths/mesa.path";
import userPath from "./paths/user.path";

const paths={
    denunciaPath:'/denuncia',
    fiscalizarPath:'/fiscalizar',
    mesaPath:'/mesa',
    userPath:'/user',
};
const mapRoute = new Map();

mapRoute.set(paths.denunciaPath,denunciaPath);
mapRoute.set(paths.fiscalizarPath,fiscalizarPath);
mapRoute.set(paths.mesaPath,mesaPath);
mapRoute.set(paths.userPath,userPath);

export default mapRoute