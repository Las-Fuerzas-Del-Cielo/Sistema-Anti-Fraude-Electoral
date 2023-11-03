import { districtsMock } from "../../pages/desk-data/_mock";
import { electoralSectionsMock } from "../../pages/desk-data/_mock";
import { sectionsMock } from "../../pages/desk-data/_mock";
import { municipalitiesMock } from "../../pages/desk-data/_mock";
import { establishmentsMock } from "../../pages/desk-data/_mock";
import { circuitsMock } from "../../pages/desk-data/_mock";
import { tables } from "../../pages/desk-data/_mock";

export const useSelectData = () => {
    return {
        districts: districtsMock,
        electoralSections: electoralSectionsMock,
        sections: sectionsMock,
        municipalities: municipalitiesMock,
        establishments: establishmentsMock,
        circuits: circuitsMock,
        tables: tables,
    };
}