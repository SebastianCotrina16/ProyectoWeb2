const express = require('express');
const cors = require('cors');
const sql = require('mssql');
const multer = require('multer');
const upload = multer();

const app = express();
const port = 5003;

app.use(cors());
app.use(express.json());

const config = {
  user: "adminbalas@sistemabalas",
  password: "Sistemabalas!",
  server: "sistemabalas.database.windows.net",
  database: "DeteccionBalas3",
  options: {
    encrypt: true,
    trustServerCertificate: true
  }
};

async function getSettings() {
  try {
    let pool = await sql.connect(config);
    let result = await pool.request().query("SELECT NumeroDisparos FROM ExamenConfiguracion");
    return result.recordset[0].NumeroDisparos;
  } catch (err) {
    console.error(err);
    return null;
  }
}

app.get('/settings', async (req, res) => {
  let numShots = await getSettings();
  if (numShots != null) {
    res.json({ status: 'success', settings: { numShots } });
  } else {
    res.status(500).json({ status: 'error', message: 'Falla al obtener los settings' });
  }
});

app.post('/results', async (req, res) => {
  const { idUsuario } = req.body;

  if (!idUsuario) {
    return res.status(400).json({ status: 'error', message: 'IdUsuario is required' });
  }

  try {
    let pool = await sql.connect(config);

    let result = await pool.request()
      .input('IdUsuario', sql.Int, idUsuario)
      .query('SELECT AVG(Precision) as score FROM ImpactosBala WHERE IdUsuario = @IdUsuario');

    if (result.recordset.length > 0) {
      res.json({ score: result.recordset[0].score });
    } else {
      res.status(404).json({ status: 'error', message: 'No records found for the given user' });
    }
  } catch (err) {
    console.error('Error fetching results from database:', err);
    res.status(500).json({ status: 'error', message: 'Error fetching results from database' });
  }
});


app.post('/detect', upload.single('imageName'), async (req, res) => {
  const { idUsuario, idReserva } = req.body;
  const imageName = req.file.originalname;

  if (!imageName || !idUsuario || !idReserva) {
    return res.status(400).json({ status: 'error', message: 'Todos los campos son obligatorios' });
  }

  let ubicacion;
  let precision;
  let rutaImagen;
  let disparos;

  if (imageName === 'image_1.jpg') {
    ubicacion = 'Ubicación 1';
    precision = 0.95;
    rutaImagen = 'ruta/imagen_1.jpg';
    disparos = [
      { precision: 0.95, ubicacion: 'Ubicación 1', rutaImagen: 'ruta/imagen_1_disparo_1.jpg' },
      { precision: 0.90, ubicacion: 'Ubicación 1', rutaImagen: 'ruta/imagen_1_disparo_2.jpg' },
    ];
  } else if (imageName === 'image_2.jpg') {
    ubicacion = 'Ubicación 2';
    precision = 0.85;
    rutaImagen = 'ruta/imagen_2.jpg';
    disparos = [
      { precision: 0.85, ubicacion: 'Ubicación 2', rutaImagen: 'ruta/imagen_2_disparo_1.jpg' },
      { precision: 0.80, ubicacion: 'Ubicación 2', rutaImagen: 'ruta/imagen_2_disparo_2.jpg' },
    ];
  } else {
    return res.status(400).json({ status: 'error', message: 'Imagen no reconocida' });
  }

  console.log("idUsuario:", idUsuario);
  console.log("idReserva:", idReserva);
  console.log("imageName:", imageName);
  console.log("ubicacion:", ubicacion);
  console.log("precision:", precision);
  console.log("rutaImagen:", rutaImagen);
  console.log("disparos:", disparos);

  try {
    let pool = await sql.connect(config);

    

    await pool.request()
      .input('IdUsuario', sql.Int, idUsuario)
      .input('Ubicacion', sql.VarChar, ubicacion)
      .input('Precision', sql.Float, precision)
      .input('RutaImagen', sql.VarChar, rutaImagen)
      .query('INSERT INTO ImpactosBala (IdUsuario, Ubicacion, Precision, RutaImagen) VALUES (@IdUsuario, @Ubicacion, @Precision, @RutaImagen)');


    const totalImpactos = disparos.length;
    const promedioPrecision = disparos.reduce((acc, disparo) => acc + disparo.precision, 0) / totalImpactos;

    console.log("totalImpactos:", totalImpactos);
    console.log("promedioPrecision:", promedioPrecision);

    await pool.request()
      .input('IdUsuario', sql.Int, idUsuario)
      .input('TotalImpactos', sql.Int, totalImpactos)
      .input('PromedioPrecision', sql.Float, promedioPrecision)
      .input('Detalles', sql.Text, JSON.stringify(disparos))
      .query('INSERT INTO Reportes (IdUsuario, TotalImpactos, PromedioPrecision, Detalles) VALUES (@IdUsuario, @TotalImpactos, @PromedioPrecision, @Detalles)');

    await pool.request()
      .input('IdUsuario', sql.Int, idUsuario)
      .input('IdReserva', sql.Int, idReserva)
      .input('TotalDisparos', sql.Int, totalImpactos)
      .input('PrecisionPromedio', sql.Float, promedioPrecision)
      .query('INSERT INTO Practicas (IdUsuario, IdReserva, TotalDisparos, PrecisionPromedio) VALUES (@IdUsuario, @IdReserva, @TotalDisparos, @PrecisionPromedio)');

    const practicasResult = await pool.request()
      .query('SELECT TOP 1 IdPractica FROM Practicas ORDER BY IdPractica DESC');
    const idPractica = practicasResult.recordset[0].IdPractica;

    for (let i = 0; i < disparos.length; i++) {
      await pool.request()
        .input('IdPractica', sql.Int, idPractica)
        .input('DisparoNumero', sql.Int, i + 1)
        .input('Precision', sql.Float, disparos[i].precision)
        .input('Ubicacion', sql.VarChar, disparos[i].ubicacion)
        .input('RutaImagen', sql.VarChar, disparos[i].rutaImagen)
        .query('INSERT INTO DetallesPracticas (IdPractica, DisparoNumero, Precision, Ubicacion, RutaImagen) VALUES (@IdPractica, @DisparoNumero, @Precision, @Ubicacion, @RutaImagen)');
    }

    res.json({ message: 'Imagen procesada y datos insertados en la base de datos' });
  } catch (err) {
    console.error('Error al insertar datos en la base de datos:', err);
    res.status(500).json({ status: 'error', message: 'Error al insertar datos en la base de datos' });
  }
});

app.listen(port, () => {
  console.log(`Admin backend listening at http://localhost:${port}`);
});
