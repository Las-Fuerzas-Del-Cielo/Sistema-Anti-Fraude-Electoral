const fs = require("fs");
const sharp = require("sharp");

const { extractText, getCleanNumber, isValueInsideBox } = require("./helpers");

const ids = [
  "0100100004X",
  "0100100100X",
  "0100100001X",
  // --
  //   "0208300005X",
  "0208300078X",
];

(async () => {
  const output = [];
  for (const id of ids) {
    try {
      let votosEnTotal,
        votosTelegramaLLA,
        votosTelegramaUP = 0;

      // datos cargados
      const getScopeData = await fetch(
        `https://resultados.gob.ar/backend-difu/scope/data/getScopeData/${id}/1`
      );
      const scopeData = await getScopeData.json();
      const { partidos } = scopeData;
      const votosCargadosLLA = partidos.find((p) => p.code == "135");
      const votosCargadosUP = partidos.find((p) => p.code == "134");

      // telegrama cargado
      const getTiff = await fetch(
        `https://resultados.gob.ar/backend-difu/scope/data/getTiff/${id}`
      );
      const {
        encodingBinary,
        metadatos: { pollingStationCode },
      } = await getTiff.json();

      // extrae datos del telegrama (pueden contenener mas de 1 pagina)
      const buffer = Buffer.from(encodingBinary, "base64");
      const imagePage = await sharp(buffer, { page: 0 }).toBuffer();
      // fs.writeFileSync(`${id}.jpeg`, imagePage);

      // itera sobre los texto para extraer los datos
      const results = await extractText(imagePage);
      const { Blocks } = results;
      let mesaId = pollingStationCode;

      for (const block of Blocks) {
        const {
          Confidence,
          Geometry: { BoundingBox },
          Text: extractedText,
        } = block;

        // menos de 95% de confianza no es un dato
        if (Confidence < 50 || !extractedText) continue;

        // coordenadas de cada texto en string para comparativa (longs)
        const left = BoundingBox.Left;
        const top = BoundingBox.Top;

        // console.debug(
        //   [left, top],
        //   extractedText,
        //   isValueInsideBox("votosTelegramaUP", [left, top])
        // );

        // total de votos
        if (isValueInsideBox("votosEnTotal", [left, top])) {
          votosEnTotal = getCleanNumber(extractedText);
        }

        // votos LLA
        if (
          !votosTelegramaLLA &&
          isValueInsideBox("votosTelegramaLLA", [left, top])
        ) {
          votosTelegramaLLA = getCleanNumber(extractedText);
        }

        //   votos UP
        if (
          !votosTelegramaUP &&
          isValueInsideBox("votosTelegramaUP", [left, top])
        ) {
          votosTelegramaUP = getCleanNumber(extractedText);
        }
      }

      const isValid =
        votosTelegramaLLA == votosCargadosLLA.votos &&
        votosTelegramaUP == votosCargadosUP.votos;

      // output
      output.push({
        mesaId,
        total: votosEnTotal || "N/A",
        LLA: {
          telegrama: votosTelegramaLLA,
          cargados: votosCargadosLLA.votos,
        },
        UP: {
          telegrama: votosTelegramaUP,
          cargados: votosCargadosUP.votos,
        },
        valido: isValid ? "✅" : "❌",
      });
      // fs.writeFileSync("extracted.json", JSON.stringify(results));
    } catch (error) {
      console.log(error);
      output.push({
        mesaId: id,
        valido: "⚠️",
      });
    } finally {
      process.stdout.write(".");
      fs.writeFileSync("output.json", JSON.stringify(output));
    }
  }
  console.log("\n");
  console.table(output);
})();
