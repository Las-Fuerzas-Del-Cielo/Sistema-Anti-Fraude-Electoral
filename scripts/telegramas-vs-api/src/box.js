// generate coords from
// https://hermitter.github.io/Polygon-Coordinates/

const box = () => {
  // medidas del telegrama
  const WIDTH = 1700;
  const HEIGHT = 2800;

  const points = [
    { x: 1211, y: 900 },
    { x: 1413, y: 900 },
    { x: 1413, y: 965 },
    { x: 1211, y: 965 },
  ];

  const coordinates = [];
  for (const coord of points) {
    const x = parseFloat((coord.x / WIDTH).toFixed(5));
    const y = parseFloat((coord.y / HEIGHT).toFixed(5));
    coordinates.push([x, y]);
  }

  console.log(JSON.stringify(coordinates));
};
box();