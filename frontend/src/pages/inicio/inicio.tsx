import Navbar from "#/components/navbar";
import { observer } from "mobx-react"

const inicioContent = () => {
    const porcentajes = [61.05, 38.95];
    const votos = ["16,482,688", "10,517,312"];
    return (
        <div className="bg-white h-screen flex flex-col">
            <Navbar />
            <p className="font-bold text-[30px] text-[#363F45] mt-5">Resultados totales</p>
            <button className="mx-3 lg:mx-60 my-10 bg-white border-[4px] border-[#61439D] text-lg text-[#61439D]">Seleccionar filtros</button>
            <div className="lg:px-60 px-3">
                {
                    //Card Javier, VLL
                }
                <div className="flex flex-col border rounded-2xl h-[50%]">
                    <div className="flex flex-col">
                        <div className="flex flex-row justify-between mb-1">
                            <img src="src/assets/logos/fenix.png" className="m-1 w-16 h-14" alt="" />
                            <div className="flex flex-col items-end mr-5 mt-2">
                                <span className="text-[12px] text-[#64748B]">{votos[0]} votos</span>
                                <p className="font-bold uppercase text-[#61439D] ">{porcentajes[1]}%</p>
                            </div>
                        </div>
                        <div className="ml-10 mb-5">
                            <div className="w-[95%] rounded-md h-2 bg-[#CBD5E1]">
                                <div className="h-full bg-[#61439D] rounded-l" style={{ width: `${porcentajes[0]}%` }}>
                                </div>
                            </div>
                            <p className="text-[13px] font-bold uppercase text-[#61439D] flex items-start">La libertad Avanza</p>
                            <p className="text-[12px] whitespace-nowrap uppercase text-[#64748B] flex items-start">JAVIER GERARDO MILEI - VICTORIA VILLARUEL</p>
                        </div>
                    </div>
                </div>
                <div className="my-4"></div>
                {
                    //Card Massa, que asco
                }
                <div className="flex flex-col border rounded-2xl h-[50%]">
                    <div className="flex flex-col">
                        <div className="flex flex-row justify-between mb-1">
                            <svg className="m-1 w-16 h-14" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink">
                                <rect width="56" height="56" fill="url(#pattern0)" />
                                <defs>
                                    <pattern id="pattern0" patternContentUnits="objectBoundingBox" width="1" height="1">
                                        <use xlinkHref="#image0_10_4663" transform="scale(0.00465116 0.00444444)" />
                                    </pattern>
                                </defs>
                            </svg>
                            <div className="flex flex-col items-end mr-5 mt-2">
                                <span className="text-[12px] text-[#64748B]">{votos[1]} votos</span>
                                <p className="font-bold uppercase text-[#61439D] ">{porcentajes[1]}%</p>
                            </div>
                        </div>
                        <div className="ml-10 mb-5">
                            <div className="w-[95%] rounded-md h-2 bg-[#CBD5E1]">
                                <div className="h-full bg-[#61439D] rounded-l" style={{ width: `${porcentajes[1]}%` }}>
                                </div>
                            </div>
                            <p className="text-[13px] font-bold uppercase text-[#61439D] flex items-start">Union por la patria</p>
                            <p className="text-[12px] whitespace-nowrap uppercase text-[#64748B] flex items-start">Sergio tomas massa - agustin rossi</p>
                        </div>
                    </div>
                </div>
            </div>
            <div className="flex flex-col px-8 lg:px-60 mt-10">
                <div className="border border-t-1 opacity-70"></div>
                <div className="my-2">
                    <span className="text-[17px] text-[#64748B]">Total de votos</span>
                    <p className="text-[25px] font-bold uppercase text-[#61439D]">27,000,000</p>
                </div>
                <div className="border border-t-1 opacity-70"></div>
                <div className="flex flex-row justify-between mt-2 px-3">
                    <div className="flex flex-col">
                        <span className="text-[17px] text-[#64748B]">Mesas escrutadas</span>
                        <p className="text-[25px] font-bold uppercase text-[#61439D]">90.00%</p>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-[17px] text-[#64748B]">Participación</span>
                        <p className="text-[25px] font-bold uppercase text-[#61439D]">76.36%</p>
                    </div>
                </div>
            </div>
            <button className="mx-3 lg:mx-60 mt-20 bg-white border-[2px] border-[#AD3459] text-lg text-[#AD3459]">Alerta Irregularidades</button>
        </div>
    )
}

export const inicio = observer(inicioContent);
export default inicio;